-include .env
export

ECR_REPO = $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/magellan

.PHONY: build push build-push ecr-login

build:
	docker compose -f docker-compose.local.yml build

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin \
		  $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com

push: ecr-login
	docker tag magellan:latest $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

build-push: build push
