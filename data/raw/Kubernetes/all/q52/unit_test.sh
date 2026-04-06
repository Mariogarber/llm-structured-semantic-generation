kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=my-ds-w-init --timeout=60s

init_container_image=$(kubectl get ds my-ds-w-init -o=jsonpath='{.spec.template.spec.initContainers[0].image}')
pod_name=$(kubectl get pods -l name=my-ds-w-init -o jsonpath='{.items[0].metadata.name}')

[ "$init_container_image" = "busybox" ] && \
[ "$(kubectl exec $pod_name -c nginx -- cat /var/shared-data/hello.txt)" = "Hello from Init" ] && \
echo cloudeval_unit_test_passed
