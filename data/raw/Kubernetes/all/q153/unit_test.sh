kubectl delete service myservice
kubectl delete service mydb
kubectl delete pod myapp-pod
sleep 5
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=init:0/2 pods/myapp-pod --timeout=20s
kubectl get pod myapp-pod | grep "Init:0/2" && echo "Correct behavior, keep going" || { echo "Wrong behavior, stopping execution"; exit 1; }
echo "---
apiVersion: v1
kind: Service
metadata:
  name: myservice
spec:
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9376
---
apiVersion: v1
kind: Service
metadata:
  name: mydb
spec:
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9377" | kubectl apply -f -
kubectl wait --for=condition=running pods/myapp-pod --timeout=20s
kubectl get pod myapp-pod | grep "Running" && echo cloudeval_unit_test_passed
# INCLUDE: "Running"
