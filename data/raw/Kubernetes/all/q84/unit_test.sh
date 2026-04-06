kubectl apply -f labeled_code.yaml
kubectl wait deployments --all --for=condition=available --timeout=20s
# make sure:
# 3/3 deployments are Ready
kubectl get deployments | awk -v RS='' '\
/3\/3/ \
{print "cloudeval_unit_test_passed"}'