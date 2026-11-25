#!/bin/sh
# Runs a fihris command. The online image first fetches the models into HF_HOME, once per
# translator set; the offline image carries them (HF_HUB_OFFLINE=1) and skips this.
set -e
SET="${FIHRIS_TRANSLATOR_MODELS:-opus-mt}"
MARK="$HF_HOME/.fihris-models-$SET"
if [ -z "${HF_HUB_OFFLINE:-}" ] && [ ! -f "$MARK" ]; then
  mkdir -p "$HF_HOME"
  echo "fihris: downloading the models ($SET and the reranker tokenizer) into $HF_HOME" >&2
  fihris download-models --translators "$SET" > "$HF_HOME/downloaded-$SET.json"
  touch "$MARK"
fi
exec fihris "$@"
