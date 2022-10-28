for FILE in `find . -type f -name "*.ui"`; do
  NAME=`basename $FILE .ui`
  CMD="pyuic5 -o $NAME.py $NAME.ui"
  echo $CMD
  eval "$CMD"
done
