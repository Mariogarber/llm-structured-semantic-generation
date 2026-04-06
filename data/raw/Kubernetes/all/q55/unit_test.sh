kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=nginx-daemon -n kube-system --timeout=60s

pods=$(kubectl get pods -l name=nginx-daemon -n kube-system --output=jsonpath={.items..metadata.name})
kubectl logs $pods -n kube-system | grep -iq "error" && exit 1

namespace=$(kubectl get ds nginx-daemon -n kube-system -o=jsonpath='{.metadata.namespace}')
[ "$namespace" = "kube-system" ] && \
echo cloudeval_unit_test_passed
