#!/bin/sh
models=${*:-"de_DE-thorsten-high de_DE-thorsten-medium de_DE-thorsten-low de_DE-ramona-low de_DE-ramona-medium de_DE-thorsten_emotional-medium"}
piper --update-voices --data-dir voices --download-dir voices --model x 2> /dev/null
for i in $models ; do
    [ ! -e "voices/$i.onnx" ] && piper --data-dir voices --download-dir voices --model $i < /dev/null > /dev/null
done
