kubectl apply -f labeled_code.yaml
sleep 20
nodeName=$(kubectl get pods -l job-name=job-pod-failure-policy-ignore -o jsonpath='{.items[0].spec.nodeName}')
kubectl drain nodes/$nodeName --ignore-daemonsets --grace-period=0 --force
sleep 10
kubectl get job job-pod-failure-policy-ignore | grep "4/4" && { echo "wrong behavior"; exit 1; }
kubectl uncordon nodes/$nodeName
sleep 40
kubectl get job job-pod-failure-policy-ignore | grep "4/4" && echo cloudeval_unit_test_passed