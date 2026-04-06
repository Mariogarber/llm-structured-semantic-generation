kubectl label nodes minikube key-name=value-name
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=fluentd-elasticsearch --timeout=20s
pods=$(kubectl get pods -l name=fluentd-elasticsearch -o=jsonpath='{.items[*].metadata.name}')
nodeName=$(kubectl get pod $pods -o=jsonpath='{.spec.nodeName}')
labelValue=$(kubectl get node $nodeName -o=jsonpath='{.metadata.labels.key-name}')

if [[ $labelValue == "value-name" ]]; then
    echo cloudeval_unit_test_passed
fi
kubectl label node minikube key-name-
