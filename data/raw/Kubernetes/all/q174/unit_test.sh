minikube node add
kubectl label nodes minikube-m02 disktype=ssd
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/nginx-select-pod --timeout=15s
kubectl describe pod nginx-select-pod | grep "minikube-m02" && echo cloudeval_unit_test_passed
# this can also be done with nodeAffinity:requiredDuringSchedulingIgnoredDuringExecution.
minikube node delete minikube-m02