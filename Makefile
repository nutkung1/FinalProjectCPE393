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

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push origin HEAD:update