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
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new model and results"
	git push --force origin HEAD:update
