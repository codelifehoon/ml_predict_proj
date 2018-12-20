# job-dir : 작업 결과물  dir
# module-name : 실행할 python 파일의 경로
# package-path  : 실행할 python의 path

MODEL_DIR=train/output

gcloud ml-engine local train \
--job-dir $MODEL_DIR \
--module-name train.google_ml_sample \
--package-path train/  \
-- \
--verbosity DEBUG

