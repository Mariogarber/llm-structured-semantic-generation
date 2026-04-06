kubectl apply -f labeled_code.yaml
kubectl wait deployments --all --for=condition=available --timeout=20s
# make sure:
# 2/2 deployments are Ready
kubectl get deployments | awk -v RS='' '\
/2\/2/ \
{print "cloudeval_unit_test_passed"}'