install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	python -m black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
   
	echo '\n## Confusion Matrix and ROC Curve Plots' >> report.md
	echo '![Model Results](./results/model_results.png)' >> report.md
   
	cml comment create report.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git commit -am "Update with new model and results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload defante/api-mack-tac-mlops ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload defante/api-mack-tac-mlops ./model /model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload defante/api-mack-tac-mlops ./results /metrics --repo-type=space --commit-message="Sync Metrics"

deploy:	hf-login push-hub