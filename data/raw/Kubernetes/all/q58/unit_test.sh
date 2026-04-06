kubectl apply -f labeled_code.yaml

node_selector=$(kubectl get ds ssd-driver -o=jsonpath='{.spec.template.spec.nodeSelector.ssd}')
container_image=$(kubectl get ds ssd-driver -o=jsonpath='{.spec.template.spec.containers[0].image}')
[ "$node_selector" = "true" ] && \
[ "$container_image" = "nginx:latest" ] && \
echo cloudeval_unit_test_passed
