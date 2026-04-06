echo "apiVersion: apps/v1
kind: Deployment
metadata:
  name: php-apache
spec:
  selector:
    matchLabels:
      run: php-apache
  template:
    metadata:
      labels:
        run: php-apache
    spec:
      containers:
      - name: php-apache
        image: registry.k8s.io/hpa-example
        ports:
        - containerPort: 80
        resources:
          limits:
            cpu: 500m
          requests:
            cpu: 200m" | kubectl apply -f -
sleep 15
kubectl apply -f labeled_code.yaml
sleep 15
kubectl get svc
timeout -s INT 8s minikube service php-apache > bash_output.txt 2>&1
cat bash_output.txt
grep "Opening service default/php-apache in default browser..." bash_output.txt && echo cloudeval_unit_test_passed
# INCLUDE: "Opening service default/nginx-service in default browser..."
