install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
   
	echo '\n## Confusion Matrix and ROC Curve Plots' >> report.md
	echo '![Model Results](./results/model_results.png)' >> report.md
   
	cml comment create report.md