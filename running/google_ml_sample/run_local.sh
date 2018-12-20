MODEL_DIR=train/output

gcloud ml-engine local train \
--job-dir $MODEL_DIR \
--module-name train.google_ml_sample \
--package-path train/  \
-- \
--runtype local \
--verbosity DEBUG

