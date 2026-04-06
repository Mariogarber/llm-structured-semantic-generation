kubectl apply -f labeled_code.yaml
# make sure:
# 1) web-0 and web-1 are eventually running
# 2) web-1 only starts after web-0 is running
# 3) no error occurs
timeout 60 kubectl get pods -w -l app=nginx | awk -v RS='' '\
/web-0[^\n]*Running.*web-1[^\n]*Running/ && \
!/web-1.*web-0/ && \
!/Error/ \
{print "cloudeval_unit_test_passed"}'