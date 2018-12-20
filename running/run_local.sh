MODEL_DIR=wordcomplete/output

gcloud ml-engine local train \
--job-dir $MODEL_DIR \
--module-name wordcomplete.wordcomplete_word_learning \
--package-path wordcomplete/  \
-- \
--runtype local \
--verbosity DEBUG

