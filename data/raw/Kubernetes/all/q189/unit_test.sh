kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/redis --timeout=20s
kubectl exec -it redis -- /bin/bash -c "cd /data/redis/ && echo Hello > test-file && kill 1 && exit" 
kubectl wait --for=condition=running pods/redis --timeout=20s
kubectl exec -it redis -- /bin/bash -c "cd /data/redis/ && ls && exit" | grep "test-file" && echo cloudeval_unit_test_passed
# INCLUDE: "test-file"
