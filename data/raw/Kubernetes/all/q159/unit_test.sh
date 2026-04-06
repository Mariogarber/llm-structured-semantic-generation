kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod/sysctls2 --timeout=60s
sysctl_values=$(kubectl get pod sysctls2 -o=jsonpath='{.spec.securityContext.sysctls}')

echo "$sysctl_values" | grep "kernel.shm_rmid_forced" | grep -q "0" && \
echo "$sysctl_values" | grep "net.ipv4.ip_local_port_range" | grep -q "1024 65535"  && \
echo "$sysctl_values" | grep "net.ipv4.tcp_syncookies" | grep -q "0"  && \
echo "$sysctl_values" | grep "net.ipv4.ping_group_range" | grep -q "1 0"  && \
echo "$sysctl_values" | grep "net.ipv4.ip_unprivileged_port_start" | grep -q "1024"  && \
echo cloudeval_unit_test_passed
