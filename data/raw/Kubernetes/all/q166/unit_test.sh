kubectl apply -f labeled_code.yaml
sleep 10

tab=$(printf '\t')

[ $(kubectl get pod alias-pod -o=jsonpath='{.spec.restartPolicy}') = "Never" ] && \
[ $(kubectl get pod alias-pod -o=jsonpath='{.spec.hostAliases[0].ip}') = "127.0.0.1" ] && \
[ $(kubectl get pod alias-pod -o=jsonpath='{.spec.hostAliases[1].ip}') = "10.1.2.3" ] && \
kubectl logs pod/alias-pod | grep -q "127.0.0.1${tab}foo.local${tab}bar.local" && \
kubectl logs pod/alias-pod | grep "10.1.2.3${tab}foo.remote${tab}bar.remote" && \
echo cloudeval_unit_test_passed
