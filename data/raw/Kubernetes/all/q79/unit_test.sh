echo "apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
" | kubectl apply -f -

kubectl wait deployments --all --for=condition=available --timeout=20s

kubectl apply -f labeled_code.yaml

kubectl wait deployments --all --for=condition=available --timeout=20s

# make sure:
# nginx 1.16.1 is used
kubectl describe deployments | awk -v RS='' '\
/1.16.1/ \
{print "cloudeval_unit_test_passed"}'