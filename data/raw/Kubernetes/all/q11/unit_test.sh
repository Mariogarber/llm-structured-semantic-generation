kubectl apply -f labeled_code.yaml
sleep 70
job_name=$(kubectl get jobs --sort-by=.metadata.creationTimestamp -o=jsonpath='{.items[0].metadata.name}')
pod_name=$(kubectl get pods --selector=job-name=$job_name -o=jsonpath='{.items[0].metadata.name}')

[ "$(kubectl get cj env-job -o jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].env[0].name}')" = "CRON_NAME" ] && \
[ "$(kubectl get cj env-job -o jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].env[0].value}')" = "CronTest" ] && \
[ "$(kubectl get pod $pod_name -o jsonpath='{.spec.containers[0].env[0].name}')" = "CRON_NAME" ] && \
[ "$(kubectl get pod $pod_name -o jsonpath='{.spec.containers[0].env[0].value}')" = "CronTest" ] && \
kubectl get cj | grep -q "1" && \
echo cloudeval_unit_test_passed
