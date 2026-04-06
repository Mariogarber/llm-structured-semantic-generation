kubectl label nodes minikube disktype=ssd
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=nodeselector-daemonset --timeout=20s
node_name=$(kubectl get pod -l app=nodeselector-daemonset -o jsonpath="{.items[0].spec.nodeName}")
if [ "$node_name" == "minikube" ]; then
    echo cloudeval_unit_test_passed
fi
kubectl label node minikube disktype-