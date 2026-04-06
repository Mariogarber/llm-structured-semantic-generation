echo "apiVersion: apps/v1                                                                                                     
kind: Deployment                                                                                                        
metadata:                                                                                                               
  name: nginx-deployment                                                                                                
spec:                                                                                                                   
  replicas: 3                                                                                                           
  selector:                                                                                                             
    matchLabels:                                                                                                        
      app: nginx                                                                                                        
  template:                                                                                                             
    metadata:                                                                                                           
      labels:                                                                                                           
        app: nginx                                                                                                      
    spec:                                                                                                               
      containers:                                                                                                       
      - name: nginx-container                                                                                           
        image: nginx:latest                                                                                             
        ports:                                                                                                          
        - containerPort: 80" | kubectl apply -f -
sleep 15
kubectl apply -f labeled_code.yaml
sleep 15
kubectl get svc
timeout -s INT 8s minikube service nginx-service > bash_output.txt 2>&1
cat bash_output.txt
grep "Opening service default/nginx-service in default browser..." bash_output.txt && echo cloudeval_unit_test_passed
# INCLUDE: "Opening service default/nginx-service in default browser..."
