kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/selinuxoptions1 --timeout=60s
selinux_type=$(kubectl get pod selinuxoptions1 -o=jsonpath='{.spec.containers[0].securityContext.seLinuxOptions.type}')
selinux_level=$(kubectl get pod selinuxoptions1 -o=jsonpath='{.spec.containers[0].securityContext.seLinuxOptions.level}')

[ "$selinux_type" = "container_init_t" ] && \
[ "$selinux_level" = "somevalue" ] && \
echo cloudeval_unit_test_passed
