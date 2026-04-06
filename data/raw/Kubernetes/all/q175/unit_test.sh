minikube node add
kubectl label nodes minikube-m02 disktype=ssd
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/nginx --timeout=20s
kubectl describe pod nginx | grep "Node:             minikube-m02" && echo "Correct behavior, keep going" || { echo "Wrong behavior, stopping execution"; minikube node delete minikube-m02; exit 1; }
minikube node delete minikube-m02
minikube node add   
kubectl label nodes minikube-m02 disktype-
kubectl label nodes minikube disktype=ssd
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/nginx --timeout=20s
kubectl describe pod nginx | grep "Node:             minikube-m02" && { echo "Wrong behavior, stopping execution"; exit 1; } || { echo "cloudeval_unit_test_passed"; }
minikube node delete minikube-m02