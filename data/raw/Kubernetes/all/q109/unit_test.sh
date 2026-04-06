kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod --selector=job-name=parallel-job-demo --timeout=60s
sleep 3
pod_name=$(kubectl get pods --selector=job-name=parallel-job-demo -o=jsonpath='{.items[0].metadata.name}')

kubectl logs pod/$pod_name | grep -q "Hey I will run till the job completes." && \
[ "$(kubectl get job parallel-job-demo -o jsonpath='{.spec.completions}')" = "4" ] && \
[ "$(kubectl get job parallel-job-demo -o jsonpath='{.spec.parallelism}')" = "2" ] && \
[ "$(kubectl get job parallel-job-demo -o jsonpath='{.spec.template.spec.containers[0].args[0]}')" = "25" ] && \
[ "$(kubectl get job parallel-job-demo -o jsonpath='{.metadata.labels.jobgroup}')" = "paralleldemo" ] && \
echo cloudeval_unit_test_passed
