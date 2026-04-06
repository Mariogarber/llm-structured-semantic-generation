kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=initcontainer --timeout=20s
pods=$(kubectl get pods -l app=initcontainer -o jsonpath="{.items[0].metadata.name}")
init_containers=$(kubectl get pod $pods -o=jsonpath='{.spec.initContainers[0].name}')
[ "$init_containers" == "init-myservice" ] && echo cloudeval_unit_test_passed