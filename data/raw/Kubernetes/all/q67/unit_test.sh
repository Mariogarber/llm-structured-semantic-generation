kubectl apply -f labeled_code.yaml
kubectl wait deployments --all --for=condition=available --timeout=20s
# make sure:
# 1/1 deployments are Ready
kubectl get deployments | awk -v RS='' '\
/1\/1/ \
{print "cloudeval_unit_test_passed"}'