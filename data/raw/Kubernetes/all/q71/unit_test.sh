kubectl apply -f labeled_code.yaml

kubectl get deployment wordpress-mysql -o jsonpath='{.spec.template.spec.volumes[0].persistentVolumeClaim.claimName}' | \
grep -q 'mysql-pv-claim' && \
echo cloudeval_unit_test_passed
