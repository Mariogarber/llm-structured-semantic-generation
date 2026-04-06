echo -n "admin" > ./username.txt
echo -n "1f2d1e2e67df" > ./password.txt
kubectl create secret generic user --from-file=./username.txt
kubectl create secret generic pass --from-file=./password.txt
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/test-projected-volume --timeout=20s
kubectl get pod test-projected-volume
kubectl exec -it test-projected-volume -- /bin/sh -c "ls /projected-volume/ && exit" | grep "username.txt" && kubectl exec -it test-projected-volume -- /bin/sh -c "ls /projected-volume/ && exit" | grep "password.txt"  && echo cloudeval_unit_test_passed
# INCLUDE: "Hello from Kubernetes storage"