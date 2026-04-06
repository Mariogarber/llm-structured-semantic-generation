kubectl apply -f labeled_code.yaml
sleep 65
pods=$(kubectl get pods -o=jsonpath='{.items[0].metadata.name}')
kubectl get cronjob hello -o yaml | grep "successfulJobsHistoryLimit: 0" && kubectl get cronjob hello -o yaml | grep "failedJobsHistoryLimit: 2" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/41385403/how-to-automatically-remove-completed-kubernetes-jobs-created-by-a-cronjob