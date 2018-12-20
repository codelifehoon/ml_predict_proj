OUTPUT_PATH=gs://ml-codelife-20181031/char_train_data/
REGION=us-central1
JOB_NM="word_learning_$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine jobs submit training $JOB_NM \
--job-dir $OUTPUT_PATH \
--runtime-version 1.9  \
--python-version 3.5 \
--module-name wordcomplete.wordcomplete_word_learning \
--package-path wordcomplete/  \
--region $REGION \
--scale-tier BASIC_GPU \
-- \
--verbosity DEBUG


gcloud ml-engine jobs stream-logs $JOB_NM
