cd ../.. # cd just outside the repo
CODENAME=ExperienceReplay
tar --exclude="chtc" --exclude="configs" --exclude='local' --exclude="plotting" --exclude='results' --exclude='.git' --exclude='.idea'  -czvf ${CODENAME}.tar.gz ${CODENAME}
scp ${CODENAME}.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm ${CODENAME}.tar.gz