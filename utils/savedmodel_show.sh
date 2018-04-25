set -u
saved_model_cli show --dir $1 --tag_set serve  --signature_def predict
