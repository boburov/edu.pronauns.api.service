"""Phoneme normalisation and feature-based similarity scoring.

Why this module exists
----------------------
The naive scorer compared phoneme tokens for *exact* equality. That is far too
harsh for a learner app: saying "hello" as ``h a l o`` instead of ``h ə l oʊ``
is essentially perfect speech, yet exact matching scores it 50%. Worse, the two
sides come from different tools (wav2vec2 vs espeak-ng) which spell the same
sound differently — one emits ``oʊ`` as a single token, the other as ``o ʊ``.

So we do two things:

1. **Normalise** both sides into the same, diacritic-free, one-sound-per-token
   space (:func:`normalize_phonemes`).
2. **Score with phonetic distance** instead of equality: substituting ``ə`` for
   ``a`` (neighbouring vowels) costs almost nothing, substituting ``k`` for
   ``m`` costs everything (:func:`similarity`).

Everything here is driven by IPA articulatory features, not by any single
language's spelling, so it works the same for all supported languages
(en, ru, tr, es, fr, de, ar).
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

# Suprasegmental / diacritic marks that carry no phoneme identity for scoring.
# espeak emits several of these; wav2vec2 emits stress digits (e.g. "ou5").
_STRIP_CHARS = frozenset(
    "ˈˌːˑ'-‿.|‖ʰʲʷˠˤ̥̬̪̺̻̃̊͜͡ⁿˀ"
    "0123456789"
)

# Characters that ARE phonemes but must never be merged with a neighbour.
_TIE_BARS = frozenset("͜͡")


def _strip_marks(token: str) -> str:
    """Drop stress/length/secondary-articulation marks from a phoneme token."""
    # NFD splits pre-composed forms (e.g. "ã" -> "a" + combining tilde) so the
    # combining marks can be filtered out uniformly.
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(
        ch for ch in decomposed
        if ch not in _STRIP_CHARS and not unicodedata.combining(ch)
    )


def explode(token: str) -> list[str]:
    """Split a phoneme token into single-sound units.

    wav2vec2 and espeak disagree on whether diphthongs and affricates are one
    token or two (``oʊ`` vs ``o ʊ``, ``tʃ`` vs ``t ʃ``). Exploding both sides
    into base characters removes the disagreement entirely.
    """
    clean = _strip_marks(token)
    return [ch for ch in clean if ch.strip() and ch not in _TIE_BARS]


def normalize_phonemes(text: str) -> str:
    """Normalise a phoneme string into space-separated single-sound tokens."""
    out: list[str] = []
    for token in text.split():
        out.extend(explode(token))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #
# Vowels: (height 0=close .. 3=open, backness 0=front .. 2=back, rounded 0/1)

_VOWELS: dict[str, tuple[float, float, int]] = {
    "i": (0.0, 0.0, 0), "y": (0.0, 0.0, 1),
    "ɨ": (0.0, 1.0, 0), "ʉ": (0.0, 1.0, 1),
    "ɯ": (0.0, 2.0, 0), "u": (0.0, 2.0, 1),
    "ɪ": (0.5, 0.2, 0), "ʏ": (0.5, 0.2, 1), "ʊ": (0.5, 1.8, 1),
    # Near-close central vowels the wav2vec2 vocabulary emits.
    "ᵻ": (0.5, 1.0, 0), "ᵿ": (0.5, 1.0, 1),
    "e": (1.0, 0.0, 0), "ø": (1.0, 0.0, 1),
    "ɘ": (1.0, 1.0, 0), "ɵ": (1.0, 1.0, 1),
    "ɤ": (1.0, 2.0, 0), "o": (1.0, 2.0, 1),
    "ə": (1.5, 1.0, 0), "ɚ": (1.5, 1.0, 0),
    "ɛ": (2.0, 0.0, 0), "œ": (2.0, 0.0, 1),
    "ɜ": (2.0, 1.0, 0), "ɝ": (2.0, 1.0, 0), "ɞ": (2.0, 1.0, 1),
    "ʌ": (2.0, 2.0, 0), "ɔ": (2.0, 2.0, 1),
    "æ": (2.5, 0.2, 0), "ɐ": (2.5, 1.0, 0),
    "a": (3.0, 0.3, 0), "ɶ": (3.0, 0.0, 1),
    "ɑ": (3.0, 2.0, 0), "ɒ": (3.0, 2.0, 1),
}

# Consonants: (place 0=bilabial .. 10=glottal, manner, voiced 0/1)
_PLOSIVE, _NASAL, _TRILL, _TAP, _FRIC, _APPROX, _LATERAL = (
    "plosive", "nasal", "trill", "tap", "fricative", "approximant", "lateral"
)

_CONSONANTS: dict[str, tuple[float, str, int]] = {
    # bilabial / labiodental
    "p": (0, _PLOSIVE, 0), "b": (0, _PLOSIVE, 1), "m": (0, _NASAL, 1),
    "ɸ": (0, _FRIC, 0), "β": (0, _FRIC, 1), "ʙ": (0, _TRILL, 1),
    "f": (1, _FRIC, 0), "v": (1, _FRIC, 1), "ʋ": (1, _APPROX, 1),
    "ɱ": (1, _NASAL, 1),
    # dental / alveolar
    "θ": (2, _FRIC, 0), "ð": (2, _FRIC, 1),
    "t": (3, _PLOSIVE, 0), "d": (3, _PLOSIVE, 1),
    "s": (3, _FRIC, 0), "z": (3, _FRIC, 1),
    "n": (3, _NASAL, 1), "l": (3, _LATERAL, 1), "ɫ": (3, _LATERAL, 1),
    "ɬ": (3, _FRIC, 0), "ɮ": (3, _FRIC, 1),
    "ɾ": (3, _TAP, 1), "r": (3, _TRILL, 1), "ɹ": (3, _APPROX, 1),
    # postalveolar / alveolo-palatal (Russian щ, Polish ś) / retroflex
    "ʃ": (4, _FRIC, 0), "ʒ": (4, _FRIC, 1),
    "ɕ": (4.5, _FRIC, 0), "ʑ": (4.5, _FRIC, 1),
    "ʈ": (5, _PLOSIVE, 0), "ɖ": (5, _PLOSIVE, 1),
    "ʂ": (5, _FRIC, 0), "ʐ": (5, _FRIC, 1),
    "ɳ": (5, _NASAL, 1), "ɻ": (5, _APPROX, 1), "ɽ": (5, _TAP, 1),
    "ɭ": (5, _LATERAL, 1),
    # palatal
    "c": (6, _PLOSIVE, 0), "ɟ": (6, _PLOSIVE, 1),
    "ç": (6, _FRIC, 0), "ʝ": (6, _FRIC, 1),
    "ɲ": (6, _NASAL, 1), "j": (6, _APPROX, 1), "ʎ": (6, _LATERAL, 1),
    # velar
    "k": (7, _PLOSIVE, 0), "g": (7, _PLOSIVE, 1), "ɡ": (7, _PLOSIVE, 1),
    "x": (7, _FRIC, 0), "ɣ": (7, _FRIC, 1),
    "ŋ": (7, _NASAL, 1), "ɰ": (7, _APPROX, 1), "w": (7, _APPROX, 1),
    # uvular
    "q": (8, _PLOSIVE, 0), "ɢ": (8, _PLOSIVE, 1),
    "χ": (8, _FRIC, 0), "ʁ": (8, _FRIC, 1),
    "ʀ": (8, _TRILL, 1), "ɴ": (8, _NASAL, 1),
    # pharyngeal (Arabic) / glottal
    "ħ": (9, _FRIC, 0), "ʕ": (9, _FRIC, 1),
    "ʔ": (10, _PLOSIVE, 0), "h": (10, _FRIC, 0), "ɦ": (10, _FRIC, 1),
}

# How far apart two manners of articulation are (0 = same, 1 = unrelated).
_MANNER_DISTANCE: dict[frozenset[str], float] = {
    frozenset({_PLOSIVE, _FRIC}): 0.45,
    frozenset({_PLOSIVE, _NASAL}): 0.55,
    frozenset({_PLOSIVE, _TAP}): 0.35,
    frozenset({_PLOSIVE, _TRILL}): 0.6,
    frozenset({_PLOSIVE, _APPROX}): 0.7,
    frozenset({_PLOSIVE, _LATERAL}): 0.7,
    frozenset({_FRIC, _NASAL}): 0.7,
    frozenset({_FRIC, _APPROX}): 0.35,
    frozenset({_FRIC, _LATERAL}): 0.6,
    frozenset({_FRIC, _TAP}): 0.6,
    frozenset({_FRIC, _TRILL}): 0.55,
    frozenset({_NASAL, _APPROX}): 0.6,
    frozenset({_NASAL, _LATERAL}): 0.5,
    frozenset({_NASAL, _TAP}): 0.7,
    frozenset({_NASAL, _TRILL}): 0.7,
    frozenset({_APPROX, _LATERAL}): 0.3,
    frozenset({_APPROX, _TAP}): 0.35,
    frozenset({_APPROX, _TRILL}): 0.45,
    frozenset({_LATERAL, _TAP}): 0.4,
    frozenset({_LATERAL, _TRILL}): 0.5,
    frozenset({_TAP, _TRILL}): 0.15,  # ɾ vs r — same sound in many accents
}

# ── Critical contrasts ────────────────────────────────────────────────────── #
# Pairs that the generic feature distance rates as "close" but which are, for a
# learner, the whole point of the exercise: r/l, s/sh, th/s. These are separate
# phonemes in every language we support, so confusing them must cost real
# points — otherwise the app would praise "light" when the word was "right".
_R_FAMILY = frozenset("rɹɾɻʁʀɽ")
_L_FAMILY = frozenset("lɫʎ")
_S_ALVEOLAR = frozenset("sz")
_S_POSTALVEOLAR = frozenset("ʃʒ")
_TH = frozenset("θð")
_F_FAMILY = frozenset("fv")

_CONTRASTS: tuple[tuple[frozenset[str], frozenset[str], float], ...] = (
    (_R_FAMILY, _L_FAMILY, 0.25),        # right / light
    (_S_ALVEOLAR, _S_POSTALVEOLAR, 0.30),  # sit / shit
    (_TH, _S_ALVEOLAR, 0.30),            # think / sink
    (_TH, _F_FAMILY, 0.30),              # three / free
)


# Rhotics ("r" sounds) are pronounced anywhere from alveolar tap to uvular
# fricative depending on the language and the speaker's accent — Spanish ɾ,
# French ʁ, English ɹ. Within the family that variation is not an error.
_FAMILY_MATCH: tuple[tuple[frozenset[str], float], ...] = (
    (_R_FAMILY, 0.85),
    (_L_FAMILY, 0.90),
)


# ── Per-language equivalences ─────────────────────────────────────────────── #
# espeak-ng picks ONE standard accent per language, but learners (and the
# recordings the model was trained on) legitimately use another. These pairs
# must not be counted as mistakes — they are correct speech in a large part of
# the language's world. Checked BEFORE the global contrast table, so a rule
# here can also *undo* a global penalty (e.g. Spanish θ/s).
_LANG_EQUIVALENTS: dict[str, tuple[tuple[frozenset[str], float], ...]] = {
    # Latin America says "gracias" with /s/ (seseo); espeak gives Castilian θ.
    # b/d/g are also spirantised to β/ð/ɣ between vowels — same phoneme.
    "es": (
        (frozenset("θs"), 0.95),
        (frozenset("βb"), 0.95),
        (frozenset("ðd"), 0.95),
        (frozenset("ɣɡg"), 0.95),
        (frozenset("ʎʝj"), 0.95),   # yeísmo
    ),
    # ich-Laut / ach-Laut are one phoneme; final -er is a schwa-ish vowel.
    "de": (
        (frozenset("çx"), 0.90),
        (frozenset("ɜə"), 0.95),
    ),
    # щ vs ш, and unstressed vowel reduction is already handled by the schwa rule.
    "ru": (
        (frozenset("ɕʃ"), 0.90),
        (frozenset("ʑʒ"), 0.90),
    ),
    # Modern Standard vs dialect: ظ/ذ overlap, ق is often realised as /k/ or /g/.
    "ar": (
        (frozenset("ðz"), 0.90),
        (frozenset("qk"), 0.85),
        (frozenset("ɡg"), 0.95),
    ),
    # ğ is silent/lengthening, and ł-like dark l is the same phoneme as l.
    "tr": (
        (frozenset("ɟɡg"), 0.90),
        (frozenset("ɫl"), 0.95),
    ),
}

# Languages whose writing system omits short vowels: espeak's reference has no
# vowels to align against, but the speaker of course pronounces them. Extra
# vowels in the prediction are therefore expected, not an error.
_IMPLICIT_VOWELS = frozenset({"ar"})


def _contrast_penalty(a: str, b: str, lang: str | None) -> float | None:
    """Fixed similarity for learner-critical confusions and accent families."""
    for group, score in _LANG_EQUIVALENTS.get(lang or "", ()):
        if a in group and b in group:
            return score
    for group_x, group_y, score in _CONTRASTS:
        if (a in group_x and b in group_y) or (a in group_y and b in group_x):
            return score
    for family, score in _FAMILY_MATCH:
        if a in family and b in family:
            return score
    return None


# Glide <-> close vowel: the model often hears "i" where espeak wrote "j"
# (and "u" where it wrote "w"). Phonetically these are the same articulation.
_GLIDE_VOWEL = {("j", "i"), ("j", "ɪ"), ("j", "y"), ("w", "u"), ("w", "ʊ")}

# The neutral / reduced vowels. Every language parks unstressed vowels here,
# and the two toolchains disagree about them constantly.
_SCHWA = frozenset("əɐɚɜɝɘ")

# Sounds that are routinely dropped or added without changing the word:
# schwa, glottal stop, breathy /h/. Missing one shouldn't tank the score.
_WEAK = frozenset("əɐʔhɦɪʊ")


def _manner_distance(m_a: str, m_b: str) -> float:
    if m_a == m_b:
        return 0.0
    return _MANNER_DISTANCE.get(frozenset({m_a, m_b}), 1.0)


@lru_cache(maxsize=8192)
def similarity(a: str, b: str, lang: str | None = None) -> float:
    """Phonetic similarity of two single-sound tokens, in [0, 1].

    1.0 = the same sound. Neighbouring sounds (``ə``/``a``, ``ɾ``/``r``,
    ``t``/``d``) land high; unrelated sounds land near 0. [lang] enables the
    accent equivalences of that language (see :data:`_LANG_EQUIVALENTS`).
    """
    if a == b:
        return 1.0

    contrast = _contrast_penalty(a, b, lang)
    if contrast is not None:
        return contrast

    va, vb = _VOWELS.get(a), _VOWELS.get(b)
    ca, cb = _CONSONANTS.get(a), _CONSONANTS.get(b)

    # ── vowel vs vowel: distance in the vowel quadrilateral ──
    # Vowel quality is the most accent-dependent thing in speech, so we stay
    # forgiving here: it is the consonants that decide which word was said.
    if va and vb:
        d_height = abs(va[0] - vb[0]) / 3.0
        d_back = abs(va[1] - vb[1]) / 2.0
        d_round = abs(va[2] - vb[2])
        dist = 0.50 * d_height + 0.35 * d_back + 0.15 * d_round
        score = 1.0 - dist
        # Schwa is the neutral/reduced vowel every language falls back to in
        # unstressed position — confusing it with any vowel is a non-error.
        if a in _SCHWA or b in _SCHWA:
            score = max(score, 0.80)
        # Cap at 0.95: different symbols are never a *perfect* match.
        return max(0.0, min(0.95, score))

    # ── consonant vs consonant: place + manner + voicing ──
    # Consonants carry word identity ("cat" vs "bat"), so the curve is sharp:
    # three places of articulation apart already counts as a different sound.
    if ca and cb:
        d_place = min(1.0, abs(ca[0] - cb[0]) / 3.0)
        d_manner = _manner_distance(ca[1], cb[1])
        d_voice = abs(ca[2] - cb[2])
        dist = 0.70 * d_place + 0.60 * d_manner + 0.25 * d_voice
        return max(0.0, min(0.95, 1.0 - dist))

    # ── glide vs close vowel (j~i, w~u) ──
    if (a, b) in _GLIDE_VOWEL or (b, a) in _GLIDE_VOWEL:
        return 0.85

    # Vowel vs consonant, or an unknown symbol: essentially no credit.
    return 0.0


def _insertion_credit(token: str, lang: str | None) -> float:
    """Credit for a sound present in the prediction but not the reference.

    Deliberately more forgiving than a deletion: the recogniser hallucinates
    extra phonemes on noisy phone recordings all the time, and in Arabic the
    reference simply has no short vowels to align against.
    """
    if lang in _IMPLICIT_VOWELS and token in _VOWELS:
        return 0.90
    if token in _WEAK:
        return 0.45
    return 0.25


def _deletion_credit(token: str, previous: str | None = None) -> float:
    """Credit for a reference sound the speaker did not produce.

    Strict by default: dropping a /k/ is a mispronunciation. Two exceptions:
      * routinely-elided weak sounds (schwa, glottal stop, /h/);
      * the off-glide of a diphthong — a vowel right after another vowel.
        Saying "hello" with a plain [o] instead of [oʊ] is an accent, not an
        error, and it is by far the most common source of false penalties.
    """
    if previous is not None and token in _VOWELS and previous in _VOWELS:
        return 0.75
    return 0.35 if token in _WEAK else 0.0


# A substitution below this similarity is not "an accent" — it is a different
# sound, i.e. a real pronunciation mistake.
_MISTAKE_THRESHOLD = 0.45

# A word containing N clearly wrong sounds cannot score above this, however
# well the rest was pronounced. Without the cap, "light" said for "right"
# scores 81% and the learner is told "excellent" for the wrong word.
_MISTAKE_CAPS = (0.69, 0.55, 0.40)


def alignment_score(
    predicted: str, reference: str, lang: str | None = None
) -> float:
    """Score how close `predicted` is to `reference`, in [0, 1].

    A Needleman-Wunsch style global alignment that maximises the total phonetic
    similarity of the aligned pairs, normalised by the longer sequence — so
    extra sounds hurt too, just less than missing ones. Clearly wrong sounds
    additionally cap the final score (see :data:`_MISTAKE_CAPS`).
    """
    a = predicted.split()
    b = reference.split()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    # dp[j] = (best score, mistakes made) aligning a[:i] with b[:j]
    # Deletion credit depends on the preceding reference sound (diphthongs).
    del_credit = [
        _deletion_credit(b[j], b[j - 1] if j else None) for j in range(m)
    ]

    prev: list[tuple[float, int]] = [(0.0, 0)] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = (prev[j - 1][0] + del_credit[j - 1], prev[j - 1][1])

    for i in range(1, n + 1):
        curr: list[tuple[float, int]] = [
            (prev[0][0] + _insertion_credit(a[i - 1], lang), prev[0][1])
        ] + [(0.0, 0)] * m
        for j in range(1, m + 1):
            sim = similarity(a[i - 1], b[j - 1], lang)
            sub = (
                prev[j - 1][0] + sim,
                prev[j - 1][1] + (1 if sim < _MISTAKE_THRESHOLD else 0),
            )
            ins = (prev[j][0] + _insertion_credit(a[i - 1], lang), prev[j][1])
            dele = (curr[j - 1][0] + del_credit[j - 1], curr[j - 1][1])
            curr[j] = max(sub, ins, dele, key=lambda pair: pair[0])
        prev = curr

    total, mistakes = prev[m]
    # Arabic: the unvocalised reference is shorter than real speech, so the
    # prediction legitimately runs longer — don't punish that length.
    denominator = m if lang in _IMPLICIT_VOWELS else max(n, m)
    score = max(0.0, min(1.0, total / float(max(1, denominator))))

    if mistakes:
        score = min(score, _MISTAKE_CAPS[min(mistakes, len(_MISTAKE_CAPS)) - 1])
    return score
