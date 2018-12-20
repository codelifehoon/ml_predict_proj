OUTPUT_PATH=gs://ml-codelife-20181219-data/train_data/
REGION=us-central1
JOB_NM="google_ml_sample$(date +%Y%m%d_%H%M%S)"

gcloud ml-engine jobs submit training $JOB_NM \
--job-dir $OUTPUT_PATH \
--runtime-version 1.9  \
--python-version 3.5 \
--module-name train.google_ml_sample \
--package-path train/  \
--region $REGION \
--scale-tier BASIC_GPU \
-- \
--verbosity DEBUG

gcloud ml-engine jobs stream-logs $JOB_NM
