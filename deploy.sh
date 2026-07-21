#!/bin/bash
# speech.sevenedu.org — talaffuz xizmatini GitHub'dagi oxirgi versiyaga yangilash.
# Yangi pip paketi kerak emas (numpy allaqachon bor), shuning uchun deploy tez.
set -eu

REPO=https://github.com/boburov/edu.pronauns.api.service.git

echo "── 1. Xizmat papkasini topish ────────────────────────────"
APP=$(ls -d /opt/*ronun* /root/*ronun* /srv/*ronun* /home/*/*ronun* \
        /opt/*peech* /root/*peech* 2>/dev/null | head -1 || true)

if [ -z "${APP:-}" ]; then
  APP=/opt/pronunciation
  echo "topilmadi → klonlanmoqda: $APP"
  git clone "$REPO" "$APP"
else
  echo "topildi: $APP"
  cd "$APP"
  if [ -d .git ]; then
    git remote set-url origin "$REPO" 2>/dev/null || git remote add origin "$REPO"
    git fetch origin main
    git reset --hard origin/main
  else
    echo "git repo emas — fayllar tarball orqali yangilanmoqda"
    curl -sL "${REPO%.git}/archive/refs/heads/main.tar.gz" \
      | tar xz --strip-components=1
  fi
fi

echo "── 2. Python muhiti ──────────────────────────────────────"
[ -d "$APP/.venv" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" -q install -r "$APP/requirements.txt"

echo "── 3. Xizmatni qayta ishga tushirish ─────────────────────"
UNIT=$(systemctl list-unit-files --type=service --no-legend 2>/dev/null \
       | awk '{print $1}' | grep -iE "ronun|peech|assess" | head -1 || true)

if [ -n "${UNIT:-}" ]; then
  echo "systemd unit: $UNIT"
  systemctl restart "$UNIT"
  sleep 12
  systemctl --no-pager --lines=5 status "$UNIT" || true
else
  echo "systemd unit topilmadi — jarayon qo'lda qayta ishga tushirilmoqda"
  pkill -f "uvicorn app:app" || true
  sleep 2
  cd "$APP"
  nohup .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000 \
    > /var/log/pronunciation.log 2>&1 &
  sleep 15
fi

echo "── 4. Tekshiruv ──────────────────────────────────────────"
for i in $(seq 1 20); do
  H=$(curl -s -m 5 http://127.0.0.1:8000/health || true)
  case "$H" in *'"model_loaded":true'*) echo "OK: $H"; break;; esac
  sleep 5
done

echo
echo "── 5. Yangi kod ishlayaptimi (sukut → tushunarli xato kutiladi) ──"
head -c 200000 /dev/zero > /tmp/z.raw
ffmpeg -loglevel error -y -f s16le -ar 16000 -ac 1 -i /tmp/z.raw /tmp/silence.wav
curl -s -m 60 -X POST http://127.0.0.1:8000/assess \
  -F "audio=@/tmp/silence.wav" -F "word=hello" -F "language=en-us"
echo
echo '↑ "code":"too_quiet" chiqsa — yangi versiya ishlayapti.'
echo '  "accuracy":0.0 chiqsa — eski kod qolgan.'
