install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	echo '```' >> report.md
	cat ./Results/metrics.txt >> report.md
	echo '```' >> report.md
	
	cml comment create report.md