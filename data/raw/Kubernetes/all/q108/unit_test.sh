kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod --selector=job-name=simple-job-demo --timeout=60s
sleep 3
pod_name=$(kubectl get pods --selector=job-name=simple-job-demo -o=jsonpath='{.items[0].metadata.name}')

kubectl logs pod/$pod_name | grep -q "Hey I will run till the job completes." && \
[ "$(kubectl get job simple-job-demo -o jsonpath='{.spec.template.spec.containers[0].args[0]}')" = "50" ] && \
[ "$(kubectl get job simple-job-demo -o jsonpath='{.metadata.labels.jobgroup}')" = "simpledemo" ] && \
echo cloudeval_unit_test_passed
