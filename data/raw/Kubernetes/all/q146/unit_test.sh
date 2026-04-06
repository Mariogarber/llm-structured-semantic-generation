kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/env-pod --timeout=60s
ENV=$(kubectl get pod env-pod -o=jsonpath='{.spec.containers[0].envFrom}')

echo $ENV | grep -q 'my-config' && \
kubectl logs pod/env-pod | grep -q 'value1' &&\
kubectl logs pod/env-pod | grep -q 'value2' &&\
echo cloudeval_unit_test_passed
