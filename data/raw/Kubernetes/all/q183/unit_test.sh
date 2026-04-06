kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/security-context-demo-2 --timeout=20s
kubectl exec -it security-context-demo-2 -- sh -c "cd /data/demo && id && exit" | egrep "groups=.*1234" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/43544370/kubernetes-how-to-set-volumemount-user-group-and-file-permissions