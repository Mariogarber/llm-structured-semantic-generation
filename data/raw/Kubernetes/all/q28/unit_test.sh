# Liveness Probe
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-liveness-probe --timeout=20s
sleep 30
kubectl describe pod ds-liveness-probe | grep "can't open '/tmp/healthy'" && echo cloudeval_unit_test_passed
