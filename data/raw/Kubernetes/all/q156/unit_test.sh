kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/selinuxoptions2 --timeout=60s
container_selinux_type=$(kubectl get pod selinuxoptions2 -o=jsonpath='{.spec.containers[0].securityContext.seLinuxOptions.type}')
initcontainer_selinux_type=$(kubectl get pod selinuxoptions2 -o=jsonpath='{.spec.initContainers[0].securityContext.seLinuxOptions.type}')
pod_selinux_type=$(kubectl get pod selinuxoptions2 -o=jsonpath='{.spec.securityContext.seLinuxOptions.type}')

[ "$container_selinux_type" = "container_init_t" ] && \
[ "$initcontainer_selinux_type" = "container_kvm_t" ] && \
[ "$pod_selinux_type" = "container_t" ] && \
echo cloudeval_unit_test_passed
