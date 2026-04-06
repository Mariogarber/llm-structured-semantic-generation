kubectl apply -f labeled_code.yaml
sleep 10

podName=$(kubectl get pods -l job-name=failjob-config -o jsonpath='{.items[0].metadata.name}')
kubectl patch pod $podName --subresource=status -p '{
  "status": {
    "conditions": [
      {
        "type": "ConfigIssue",
        "status": "True",
        "reason": "NonExistingImage",
        "lastTransitionTime": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
      }
    ]
  }
}'
kubectl delete pods/$podName

kubectl get job failjob-config -o=jsonpath='{.status.conditions[*].message}' | grep -q "ConfigIssue" && \
kubectl get job failjob-config -o=jsonpath='{.status.conditions[*].reason}' | grep -q "PodFailurePolicy" && \
echo cloudeval_unit_test_passed
