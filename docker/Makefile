MODEL_IMAGE=tf-airbnb-pricing-mx
MODEL_TAG=v_20240107184239

STREAMLIT_IMAGE=streamlit-app-airbnb-pricing-mx

build-image-model:
	@docker build -t ${MODEL_IMAGE}:${MODEL_TAG} -f image-model.dockerfile .

run-image-model-container:
	@docker run -it --rm -p 8080:8080 ${MODEL_IMAGE}:${MODEL_TAG}

run-complete-service:
	@docker build -t ${STREAMLIT_IMAGE} -f image-streamlit.dockerfile .
	@docker-compose up -d

shutdown:
	@docker-compose down