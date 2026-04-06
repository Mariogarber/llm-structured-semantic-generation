kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/goproxy --timeout=15s
kubectl get pod etcd-with-grpc -o yaml | grep "restartCount: 0" && kubectl get pod etcd-with-grpc -o yaml | grep "livenessProbe:" && kubectl get pod etcd-with-grpc -o yaml | grep "grpc:
        port: 2379" &&  echo cloudeval_unit_test_passed
