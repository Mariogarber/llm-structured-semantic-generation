kubectl apply -f labeled_code.yaml

kubectl create serviceaccount test-sa

kubectl create clusterrolebinding test-binding --clusterrole=cluster-role-simple --serviceaccount=default:test-sa

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-clusterrole-pod
spec:
  serviceAccountName: test-sa
  containers:
  - name: test-container
    image: busybox
    command:
    - "sh"
    - "-c"
    - "sleep 3600"
EOF

kubectl wait --for=condition=Ready pod/test-clusterrole-pod --timeout=60s

if kubectl exec test-clusterrole-pod -- sh -c "echo" > /dev/null; then
    echo "Pod test-clusterrole-pod is running."
    if kubectl exec test-clusterrole-pod -- sh -c "ls /var/run/secrets/kubernetes.io/serviceaccount/" | grep -E "namespace|token|ca.crt"; then
        if kubectl get clusterrole cluster-role-simple | grep "CREATED AT"; then
            echo cloudeval_unit_test_passed
        fi
    else
        echo "Test failed"
    fi
else
    echo "Test failed"
fi

kubectl delete pod test-clusterrole-pod
kubectl delete clusterrolebinding test-binding
kubectl delete serviceaccount test-sa
kubectl delete -f labeled_code.yaml
