kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=nginx-daemon --timeout=60s

pods=$(kubectl get pods -l name=nginx-daemon --output=jsonpath={.items..metadata.name})
kubectl logs $pods | grep -iq "error" && exit 1

tolerations=$(kubectl get ds nginx-daemon -o=jsonpath='{.spec.template.spec.tolerations[*].key}')
echo $tolerations | grep -q "node-role.kubernetes.io/master" && \
echo $tolerations | grep -q "node-role.kubernetes.io/control-plane" && \
echo cloudeval_unit_test_passed
