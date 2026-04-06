kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/security-context-demo --timeout=20s
kubectl exec -it security-context-demo -- sh -c "cd /data/demo && id && exit" | grep "uid=1000 gid=3000 groups=2000" && echo cloudeval_unit_test_passed
# there are still more test can be done, see offical doc.